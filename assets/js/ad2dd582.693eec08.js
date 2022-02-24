"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[390],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return b}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},d=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),u=l(n),b=a,f=u["".concat(c,".").concat(b)]||u[b]||p[b]||o;return n?r.createElement(f,i(i({ref:t},d),{},{components:n})):r.createElement(f,i({ref:t},d))}));function b(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=u;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var l=2;l<o;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},4999:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return d},default:function(){return u}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),i=["components"],s={},c="Gender Bias Analysis",l={unversionedId:"WebUI/bias_analysis_for_gender_bias",id:"WebUI/bias_analysis_for_gender_bias",title:"Gender Bias Analysis",description:"DataLab can be used for quantify the proportion of male words (or entities) to female words (or entities) given a  dataset.",source:"@site/docs/WebUI/4_bias_analysis_for_gender_bias.md",sourceDirName:"WebUI",slug:"/WebUI/bias_analysis_for_gender_bias",permalink:"/DataLab/docs/WebUI/bias_analysis_for_gender_bias",editUrl:"https://github.com/ExpressAI/DataLab/tree/main/docs/WebUI/4_bias_analysis_for_gender_bias.md",tags:[],version:"current",sidebarPosition:4,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Hate Speech Analysis",permalink:"/DataLab/docs/WebUI/bias_analysis_for_hate_speech"},next:{title:"Identify Dataset Artifacts",permalink:"/DataLab/docs/WebUI/bias_analysis_for_artifacts"}},d=[{value:"1. Dataset Selection",id:"1-dataset-selection",children:[],level:3},{value:"2. Choose the <code>gender bias</code> option in the drop-down box",id:"2-choose-the-gender-bias-option-in-the-drop-down-box",children:[],level:3}],p={toc:d};function u(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"gender-bias-analysis"},"Gender Bias Analysis"),(0,o.kt)("p",null,"DataLab can be used for quantify the proportion of male words (or entities) to female words (or entities) given a  dataset."),(0,o.kt)("p",null,"To perform this type of analysis:"),(0,o.kt)("h3",{id:"1-dataset-selection"},"1. Dataset Selection"),(0,o.kt)("p",null,"You just need to choose a dataset and click the right mouse button, and choose ",(0,o.kt)("inlineCode",{parentName:"p"},"analysis")," -> ",(0,o.kt)("inlineCode",{parentName:"p"},"bias"),", then you will enter into a page designed for bias analysis"),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155384702-9c7dc15b-036f-4ce4-906d-1258075dad8a.png",width:"200"}),(0,o.kt)("h3",{id:"2-choose-the-gender-bias-option-in-the-drop-down-box"},"2. Choose the ",(0,o.kt)("inlineCode",{parentName:"h3"},"gender bias")," option in the drop-down box"),(0,o.kt)("p",null,"As shown below, different colors represent the proportions of samples with male words and female words."),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155390945-6bd2fed7-beca-4cb6-b5a6-380287562da5.png",width:"600"}))}u.isMDXComponent=!0}}]);